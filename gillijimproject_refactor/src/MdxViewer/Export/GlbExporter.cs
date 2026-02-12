using System.Numerics;
using System.Text;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using SereniaBLPLib;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Scenes;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
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
    public static void ExportMdx(MdxFile mdx, string modelDir, string outputPath, IDataSource? dataSource = null)
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

            // Try to resolve texture as PNG bytes (decode BLP if needed)
            byte[]? pngBytes = ResolveMdxTexturePngBytes(mdx, geoset, modelDir, dataSource);
            if (pngBytes != null)
            {
                var img = new SharpGLTF.Memory.MemoryImage(pngBytes);
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
        ViewerLog.Trace($"  GLB: {mdx.Geosets.Count(g => g.Vertices.Count > 0)} geosets exported");
    }

    /// <summary>
    /// Export a WMO (v14 parsed data) to GLB with per-group meshes and material colors.
    /// </summary>
    public static void ExportWmo(WmoV14ToV17Converter.WmoV14Data wmo, string modelDir, string outputPath, IDataSource? dataSource = null)
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
                byte[]? pngBytes = ResolveWmoTexturePngBytes(wmoMat.Texture1Name, modelDir, dataSource);
                if (pngBytes != null)
                {
                    var img = new SharpGLTF.Memory.MemoryImage(pngBytes);
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
        ViewerLog.Trace($"  GLB: {wmo.Groups.Count} groups, {wmo.Materials.Count} materials exported");
    }

    /// <summary>
    /// Export a WMO with all groups, materials, and doodad MDX models combined into a single GLB.
    /// Doodad set 0 (DefaultGlobal) is exported. Each doodad MDX is added as a mesh instance
    /// with its placement transform (Scale × Rotation × Translation), all in Y-up glTF space.
    /// </summary>
    public static void ExportWmoWithDoodads(
        WmoV14ToV17Converter.WmoV14Data wmo, string modelDir, string outputPath,
        IDataSource? dataSource = null, int doodadSetIndex = 0)
    {
        var scene = new SceneBuilder();

        // --- 1. Build WMO material lookup ---
        var materials = new Dictionary<byte, MaterialBuilder>();
        for (int m = 0; m < wmo.Materials.Count; m++)
        {
            var wmoMat = wmo.Materials[m];
            var matBuilder = new MaterialBuilder($"wmo_mat_{m}");
            matBuilder.WithDoubleSide(true);

            if (!string.IsNullOrEmpty(wmoMat.Texture1Name))
            {
                byte[]? pngBytes = ResolveWmoTexturePngBytes(wmoMat.Texture1Name, modelDir, dataSource);
                if (pngBytes != null)
                {
                    var img = new SharpGLTF.Memory.MemoryImage(pngBytes);
                    matBuilder.WithBaseColor(img);
                }
                else
                {
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

        var defaultMat = new MaterialBuilder("wmo_default");
        defaultMat.WithBaseColor(new Vector4(0.6f, 0.6f, 0.6f, 1.0f));
        defaultMat.WithDoubleSide(true);

        // --- 2. Add WMO group meshes ---
        for (int gi = 0; gi < wmo.Groups.Count; gi++)
        {
            var group = wmo.Groups[gi];
            if (group.Vertices.Count == 0 || group.Indices.Count == 0)
                continue;

            string groupName = group.Name ?? $"group_{gi}";
            var mesh = new MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>(groupName);

            var normals = GenerateNormals(group);
            bool hasUVs = group.UVs.Count == group.Vertices.Count;

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

        // --- 3. Add doodad MDX models from the active doodad set ---
        int doodadsLoaded = 0, doodadsFailed = 0;
        var doodadMeshCache = new Dictionary<string, MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>?>(
            StringComparer.OrdinalIgnoreCase);

        if (wmo.DoodadSets.Count > 0 && wmo.DoodadDefs.Count > 0 && dataSource != null)
        {
            int setIdx = Math.Clamp(doodadSetIndex, 0, wmo.DoodadSets.Count - 1);
            var set = wmo.DoodadSets[setIdx];
            ViewerLog.Trace($"[GlbExporter] Exporting DoodadSet [{setIdx}] \"{set.Name}\": {set.Count} doodads");

            for (uint i = set.StartIndex; i < set.StartIndex + set.Count && i < (uint)wmo.DoodadDefs.Count; i++)
            {
                var def = wmo.DoodadDefs[(int)i];
                string modelPath = GetDoodadName(wmo.DoodadNamesRaw, def.NameIndex);

                if (string.IsNullOrEmpty(modelPath))
                {
                    doodadsFailed++;
                    continue;
                }

                string normalized = modelPath.Replace('/', '\\').ToLowerInvariant();

                // Get or build mesh for this doodad model
                if (!doodadMeshCache.TryGetValue(normalized, out var doodadMesh))
                {
                    doodadMesh = LoadDoodadMesh(modelPath, modelDir, dataSource);
                    doodadMeshCache[normalized] = doodadMesh;
                }

                if (doodadMesh == null)
                {
                    doodadsFailed++;
                    continue;
                }

                // Build placement transform: Scale × Rotation × Translation
                // Then convert from WoW Z-up to glTF Y-up
                var transform = BuildDoodadTransformYup(def.Position, def.Orientation, def.Scale);
                scene.AddRigidMesh(doodadMesh, transform);
                doodadsLoaded++;
            }
        }

        var glbModel = scene.ToGltf2();
        glbModel.SaveGLB(outputPath);
        ViewerLog.Trace($"[GlbExporter] WMO GLB: {wmo.Groups.Count} groups, {wmo.Materials.Count} materials, " +
                          $"{doodadsLoaded} doodads ({doodadMeshCache.Count} unique models), {doodadsFailed} failed → {outputPath}");
    }

    /// <summary>
    /// Build a Y-up placement transform for a WMO doodad.
    /// WoW doodad def: Position(Z-up), Orientation(quaternion in Z-up), Scale.
    /// glTF expects Y-up, so we convert: (X,Y,Z) → (X,Z,-Y) for positions,
    /// and (qx,qy,qz,qw) → (qx,qz,-qy,qw) for quaternions.
    /// </summary>
    private static Matrix4x4 BuildDoodadTransformYup(Vector3 pos, Quaternion rot, float scale)
    {
        // Convert position from Z-up to Y-up
        var yupPos = new Vector3(pos.X, pos.Z, -pos.Y);

        // Convert quaternion from Z-up to Y-up
        var yupRot = new Quaternion(rot.X, rot.Z, -rot.Y, rot.W);

        return Matrix4x4.CreateScale(scale)
             * Matrix4x4.CreateFromQuaternion(yupRot)
             * Matrix4x4.CreateTranslation(yupPos);
    }

    /// <summary>
    /// Resolve a doodad name from the raw MODN string table.
    /// </summary>
    private static string GetDoodadName(byte[] doodadNamesRaw, uint nameOffset)
    {
        if (doodadNamesRaw.Length == 0 || nameOffset >= doodadNamesRaw.Length) return "";
        int end = (int)nameOffset;
        while (end < doodadNamesRaw.Length && doodadNamesRaw[end] != 0)
            end++;
        if (end == (int)nameOffset) return "";
        return Encoding.UTF8.GetString(doodadNamesRaw, (int)nameOffset, end - (int)nameOffset);
    }

    /// <summary>
    /// Load an MDX doodad model from the data source and build a SharpGLTF mesh.
    /// Returns null if the model can't be loaded.
    /// </summary>
    private static MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>? LoadDoodadMesh(
        string modelPath, string modelDir, IDataSource dataSource)
    {
        try
        {
            byte[]? mdxData = dataSource.ReadFile(modelPath);

            // Try MDL/MDX swap if not found
            if (mdxData == null)
            {
                string ext = Path.GetExtension(modelPath).ToLowerInvariant();
                string altPath = ext == ".mdl"
                    ? Path.ChangeExtension(modelPath, ".mdx")
                    : ext == ".mdx"
                        ? Path.ChangeExtension(modelPath, ".mdl")
                        : modelPath;
                if (altPath != modelPath)
                    mdxData = dataSource.ReadFile(altPath);
            }

            if (mdxData == null || mdxData.Length == 0) return null;

            using var ms = new MemoryStream(mdxData);
            var mdx = MdxFile.Load(ms);

            string doodadName = Path.GetFileNameWithoutExtension(modelPath);
            var mesh = new MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>($"doodad_{doodadName}");

            for (int gi = 0; gi < mdx.Geosets.Count; gi++)
            {
                var geoset = mdx.Geosets[gi];
                if (geoset.Vertices.Count == 0 || geoset.Indices.Count == 0)
                    continue;

                // Resolve material/texture for this geoset
                var matBuilder = new MaterialBuilder($"doodad_{doodadName}_mat_{gi}");
                matBuilder.WithDoubleSide(true);

                byte[]? pngBytes = ResolveMdxTexturePngBytes(mdx, geoset, modelDir, dataSource);
                if (pngBytes != null)
                {
                    var img = new SharpGLTF.Memory.MemoryImage(pngBytes);
                    matBuilder.WithBaseColor(img);
                }
                else
                {
                    matBuilder.WithBaseColor(new Vector4(0.7f, 0.7f, 0.7f, 1.0f));
                }

                var prim = mesh.UsePrimitive(matBuilder);

                bool hasNormals = geoset.Normals.Count == geoset.Vertices.Count;
                bool hasUVs = geoset.TexCoords.Count == geoset.Vertices.Count;

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
            }

            ViewerLog.Trace($"[GlbExporter] Doodad loaded: {doodadName} ({mdx.Geosets.Count} geosets)");
            return mesh;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[GlbExporter] Doodad load failed: {modelPath} — {ex.Message}");
            return null;
        }
    }

    /// <summary>Z-up (WoW) to Y-up (glTF): (X,Y,Z) → (X,Z,-Y)</summary>
    private static Vector3 ZupToYup(float x, float y, float z) => new Vector3(x, z, -y);

    private static VERTEX MakeVertex(MdlGeoset geoset, int idx, bool hasNormals, bool hasUVs)
    {
        var pos = geoset.Vertices[idx];
        var position = ZupToYup(pos.X, pos.Y, pos.Z);

        var normal = hasNormals
            ? ZupToYup(geoset.Normals[idx].X, geoset.Normals[idx].Y, geoset.Normals[idx].Z)
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
        var position = ZupToYup(pos.X, pos.Y, pos.Z);
        var normal = idx < normals.Count
            ? ZupToYup(normals[idx].X, normals[idx].Y, normals[idx].Z)
            : Vector3.UnitY;
        var uv = hasUVs ? group.UVs[idx] : Vector2.Zero;

        return new VERTEX(
            new VertexPositionNormal(position, normal),
            new VertexTexture1(uv)
        );
    }

    /// <summary>
    /// Resolve an MDX geoset's texture to PNG bytes, decoding BLP from MPQ if needed.
    /// </summary>
    private static byte[]? ResolveMdxTexturePngBytes(MdxFile mdx, MdlGeoset geoset, string modelDir, IDataSource? dataSource)
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

        string fileName = Path.GetFileName(tex.Path);

        // 1. Try local PNG file
        string pngPath = Path.Combine(modelDir, Path.ChangeExtension(fileName, ".png"));
        if (File.Exists(pngPath))
            return File.ReadAllBytes(pngPath);

        // 2. Try local BLP file → decode to PNG
        string blpPath = Path.Combine(modelDir, Path.ChangeExtension(fileName, ".blp"));
        if (File.Exists(blpPath))
            return BlpToPngBytes(File.ReadAllBytes(blpPath));

        // 3. Try MPQ data source
        if (dataSource != null)
        {
            byte[]? blpData = dataSource.ReadFile(tex.Path);
            if (blpData != null && blpData.Length > 0)
                return BlpToPngBytes(blpData);
        }

        return null;
    }

    /// <summary>
    /// Resolve a WMO material texture to PNG bytes, decoding BLP from MPQ if needed.
    /// </summary>
    private static byte[]? ResolveWmoTexturePngBytes(string textureName, string modelDir, IDataSource? dataSource)
    {
        string fileName = Path.GetFileName(textureName);

        // 1. Try local PNG file
        string pngPath = Path.Combine(modelDir, Path.ChangeExtension(fileName, ".png"));
        if (File.Exists(pngPath))
            return File.ReadAllBytes(pngPath);

        // 2. Try local BLP file → decode to PNG
        string blpPath = Path.Combine(modelDir, Path.ChangeExtension(fileName, ".blp"));
        if (File.Exists(blpPath))
            return BlpToPngBytes(File.ReadAllBytes(blpPath));

        // 3. Try MPQ data source
        if (dataSource != null)
        {
            byte[]? blpData = dataSource.ReadFile(textureName);
            if (blpData != null && blpData.Length > 0)
                return BlpToPngBytes(blpData);
        }

        return null;
    }

    /// <summary>
    /// Decode BLP texture data to PNG bytes for embedding in GLB.
    /// </summary>
    private static byte[]? BlpToPngBytes(byte[] blpData)
    {
        try
        {
            using var ms = new MemoryStream(blpData);
            var blp = new BlpFile(ms);
            var bmp = blp.GetBitmap(0);
            int w = bmp.Width, h = bmp.Height;

            // Convert System.Drawing.Bitmap BGRA → RGBA bytes
            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var lockBits = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            byte[] pixels = new byte[w * h * 4];
            System.Runtime.InteropServices.Marshal.Copy(lockBits.Scan0, pixels, 0, pixels.Length);
            bmp.UnlockBits(lockBits);
            bmp.Dispose();

            // BGRA → RGBA swap
            for (int i = 0; i < pixels.Length; i += 4)
            {
                (pixels[i], pixels[i + 2]) = (pixels[i + 2], pixels[i]);
            }

            // Encode as PNG using ImageSharp
            using var image = SixLabors.ImageSharp.Image.LoadPixelData<Rgba32>(pixels, w, h);
            using var output = new MemoryStream();
            image.SaveAsPng(output);
            return output.ToArray();
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[GlbExporter] BLP→PNG decode failed: {ex.Message}");
            return null;
        }
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
