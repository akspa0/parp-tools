using System.Numerics;
using MdxLTool.Formats.Mdx;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Scenes;
using WoWMapConverter.Core.Converters;

namespace MdxViewer.Export;

using VERTEX = VertexBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>;

/// <summary>
/// Exports MDX and WMO models to GLB (binary glTF) with materials.
/// Uses SharpGLTF.Toolkit for spec-compliant output.
/// </summary>
public static class GlbExporter
{
    /// <summary>
    /// Export an MDX model to GLB with per-geoset materials.
    /// </summary>
    public static void ExportMdx(MdxFile mdx, string modelDir, string outputPath)
    {
        var scene = new SceneBuilder();

        for (int i = 0; i < mdx.Geosets.Count; i++)
        {
            var geoset = mdx.Geosets[i];
            if (geoset.Vertices.Count == 0 || geoset.Indices.Count == 0)
                continue;

            // Resolve material
            var matBuilder = new MaterialBuilder($"mat_{i}");
            matBuilder.WithDoubleSide(true);

            // Try to find texture PNG
            string? texPng = ResolveMdxTexturePng(mdx, geoset, modelDir);
            if (texPng != null && File.Exists(texPng))
            {
                var imgBytes = File.ReadAllBytes(texPng);
                var img = new SharpGLTF.Memory.MemoryImage(imgBytes);
                matBuilder.WithBaseColor(img);
            }
            else
            {
                matBuilder.WithBaseColor(new Vector4(0.8f, 0.8f, 0.8f, 1.0f));
            }

            var mesh = new MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>($"geoset_{i}");
            var prim = mesh.UsePrimitive(matBuilder);

            bool hasNormals = geoset.Normals.Count == geoset.Vertices.Count;
            bool hasUVs = geoset.TexCoords.Count == geoset.Vertices.Count;

            // Build triangles
            for (int t = 0; t + 2 < geoset.Indices.Count; t += 3)
            {
                int i0 = geoset.Indices[t];
                int i1 = geoset.Indices[t + 1];
                int i2 = geoset.Indices[t + 2];

                if (i0 >= geoset.Vertices.Count || i1 >= geoset.Vertices.Count || i2 >= geoset.Vertices.Count)
                    continue;

                var v0 = MakeVertex(geoset, i0, hasNormals, hasUVs);
                var v1 = MakeVertex(geoset, i1, hasNormals, hasUVs);
                var v2 = MakeVertex(geoset, i2, hasNormals, hasUVs);

                prim.AddTriangle(v0, v1, v2);
            }

            scene.AddRigidMesh(mesh, Matrix4x4.Identity);
        }

        var model = scene.ToGltf2();
        model.SaveGLB(outputPath);
        Console.WriteLine($"  GLB: {mdx.Geosets.Count(g => g.Vertices.Count > 0)} geosets exported");
    }

    /// <summary>
    /// Export a WMO (v14 parsed data) to GLB with per-group meshes and material colors.
    /// </summary>
    public static void ExportWmo(WmoV14ToV17Converter.WmoV14Data wmo, string modelDir, string outputPath)
    {
        var scene = new SceneBuilder();

        // Build material lookup from WMO materials
        var materials = new Dictionary<byte, MaterialBuilder>();
        for (int m = 0; m < wmo.Materials.Count; m++)
        {
            var wmoMat = wmo.Materials[m];
            var matBuilder = new MaterialBuilder($"wmo_mat_{m}");
            matBuilder.WithDoubleSide(true);

            // Try to load texture
            if (!string.IsNullOrEmpty(wmoMat.Texture1Name))
            {
                string pngName = Path.ChangeExtension(Path.GetFileName(wmoMat.Texture1Name), ".png");
                string pngPath = Path.Combine(modelDir, pngName);
                if (File.Exists(pngPath))
                {
                    var imgBytes = File.ReadAllBytes(pngPath);
                    var img = new SharpGLTF.Memory.MemoryImage(imgBytes);
                    matBuilder.WithBaseColor(img);
                }
                else
                {
                    // Assign a distinct color per material
                    float r = ((m * 67 + 13) % 255) / 255f;
                    float g = ((m * 131 + 7) % 255) / 255f;
                    float b = ((m * 43 + 29) % 255) / 255f;
                    matBuilder.WithBaseColor(new Vector4(r, g, b, 1.0f));
                }
            }
            else
            {
                float r = ((m * 67 + 13) % 255) / 255f;
                float g = ((m * 131 + 7) % 255) / 255f;
                float b = ((m * 43 + 29) % 255) / 255f;
                matBuilder.WithBaseColor(new Vector4(r, g, b, 1.0f));
            }

            materials[(byte)m] = matBuilder;
        }

        // Default material for unmapped faces
        var defaultMat = new MaterialBuilder("wmo_default");
        defaultMat.WithBaseColor(new Vector4(0.6f, 0.6f, 0.6f, 1.0f));
        defaultMat.WithDoubleSide(true);

        for (int gi = 0; gi < wmo.Groups.Count; gi++)
        {
            var group = wmo.Groups[gi];
            if (group.Vertices.Count == 0 || group.Indices.Count == 0)
                continue;

            string groupName = group.Name ?? $"group_{gi}";
            var mesh = new MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>(groupName);

            // Generate normals
            var normals = GenerateNormals(group);
            bool hasUVs = group.UVs.Count == group.Vertices.Count;

            // Build triangles grouped by face material
            int faceCount = group.Indices.Count / 3;
            for (int f = 0; f < faceCount; f++)
            {
                int idx = f * 3;
                if (idx + 2 >= group.Indices.Count) break;

                int i0 = group.Indices[idx];
                int i1 = group.Indices[idx + 1];
                int i2 = group.Indices[idx + 2];

                if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
                    continue;

                // Get material for this face
                byte matId = f < group.FaceMaterials.Count ? group.FaceMaterials[f] : (byte)0;
                var mat = materials.GetValueOrDefault(matId, defaultMat);
                var prim = mesh.UsePrimitive(mat);

                var v0 = MakeWmoVertex(group, i0, normals, hasUVs);
                var v1 = MakeWmoVertex(group, i1, normals, hasUVs);
                var v2 = MakeWmoVertex(group, i2, normals, hasUVs);

                prim.AddTriangle(v0, v1, v2);
            }

            scene.AddRigidMesh(mesh, Matrix4x4.Identity);
        }

        var model = scene.ToGltf2();
        model.SaveGLB(outputPath);
        Console.WriteLine($"  GLB: {wmo.Groups.Count} groups, {wmo.Materials.Count} materials exported");
    }

    private static VERTEX MakeVertex(MdlGeoset geoset, int idx, bool hasNormals, bool hasUVs)
    {
        var pos = geoset.Vertices[idx];
        var position = new Vector3(-pos.X, pos.Y, pos.Z); // Flip X for WoW coords

        var normal = hasNormals
            ? new Vector3(-geoset.Normals[idx].X, geoset.Normals[idx].Y, geoset.Normals[idx].Z)
            : Vector3.UnitY;

        var uv = hasUVs
            ? new Vector2(geoset.TexCoords[idx].U, geoset.TexCoords[idx].V)
            : Vector2.Zero;

        return new VERTEX(
            new VertexPositionNormal(position, normal),
            new VertexTexture1(uv)
        );
    }

    private static VERTEX MakeWmoVertex(
        WmoV14ToV17Converter.WmoGroupData group, int idx,
        List<Vector3> normals, bool hasUVs)
    {
        var pos = group.Vertices[idx];
        var normal = idx < normals.Count ? normals[idx] : Vector3.UnitY;
        var uv = hasUVs ? group.UVs[idx] : Vector2.Zero;

        return new VERTEX(
            new VertexPositionNormal(pos, normal),
            new VertexTexture1(uv)
        );
    }

    private static string? ResolveMdxTexturePng(MdxFile mdx, MdlGeoset geoset, string modelDir)
    {
        int texId = -1;
        if (geoset.MaterialId >= 0 && geoset.MaterialId < mdx.Materials.Count)
        {
            var mat = mdx.Materials[geoset.MaterialId];
            if (mat.Layers.Count > 0 && mat.Layers[0].TextureId >= 0 && mat.Layers[0].TextureId < mdx.Textures.Count)
                texId = mat.Layers[0].TextureId;
        }
        if (texId < 0 && mdx.Textures.Count > 0) texId = 0;
        if (texId < 0) return null;

        var tex = mdx.Textures[texId];
        if (string.IsNullOrEmpty(tex.Path)) return null;

        string pngName = Path.ChangeExtension(Path.GetFileName(tex.Path), ".png");
        string pngPath = Path.Combine(modelDir, pngName);
        return File.Exists(pngPath) ? pngPath : null;
    }

    private static List<Vector3> GenerateNormals(WmoV14ToV17Converter.WmoGroupData group)
    {
        var normals = new Vector3[group.Vertices.Count];
        for (int i = 0; i + 2 < group.Indices.Count; i += 3)
        {
            int i0 = group.Indices[i], i1 = group.Indices[i + 1], i2 = group.Indices[i + 2];
            if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
                continue;
            var e1 = group.Vertices[i1] - group.Vertices[i0];
            var e2 = group.Vertices[i2] - group.Vertices[i0];
            var n = Vector3.Cross(e1, e2);
            if (n.LengthSquared() > 0.0001f)
                n = Vector3.Normalize(n);
            else
                continue;
            normals[i0] += n;
            normals[i1] += n;
            normals[i2] += n;
        }
        return normals.Select(n => n.Length() > 0.001f ? Vector3.Normalize(n) : Vector3.UnitY).ToList();
    }
}
