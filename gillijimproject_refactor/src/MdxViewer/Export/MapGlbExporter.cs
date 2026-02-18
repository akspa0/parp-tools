using System.Numerics;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using MdxViewer.Terrain;
using SereniaBLPLib;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Scenes;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WoWMapConverter.Core.Converters;
using WoWMapConverter.Core.Services;

namespace MdxViewer.Export;

using VERTEX = VertexBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>;

public static class MapGlbExporter
{
    private static readonly Matrix4x4 ZupToYupMatrix = Matrix4x4.CreateRotationX(-MathF.PI / 2f);

    public static void ExportTile(
        TerrainManager terrain,
        IDataSource dataSource,
        Md5TranslateIndex? md5Index,
        int tileX,
        int tileY,
        string outputPath,
        bool includePlacements = true)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");

        var result = terrain.GetOrLoadTileLoadResult(tileX, tileY);

        var scene = new SceneBuilder();

        // Terrain
        if (result.Chunks.Count > 0)
        {
            var terrainMesh = BuildTerrainTileMesh(tileX, tileY, result.Chunks, terrain.MapName, dataSource, md5Index);
            if (terrainMesh != null)
            {
                scene.AddRigidMesh(terrainMesh, Matrix4x4.Identity);
            }
        }

        if (includePlacements)
        {
            AddPlacements(scene, terrain.Adapter, result, dataSource);
        }

        var model = scene.ToGltf2();
        model.SaveGLB(outputPath);

        ViewerLog.Important(ViewerLog.Category.General,
            $"[MapGlbExporter] Exported tile ({tileX},{tileY}) to {outputPath} " +
            $"(chunks={result.Chunks.Count}, mddf={result.MddfPlacements.Count}, modf={result.ModfPlacements.Count})");
    }

    private static void AddPlacements(SceneBuilder scene, ITerrainAdapter adapter, TileLoadResult result, IDataSource dataSource)
    {
        var mdxNames = adapter.MdxModelNames;
        var wmoNames = adapter.WmoModelNames;

        var mdxMeshCache = new Dictionary<string, MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>?>(StringComparer.OrdinalIgnoreCase);
        var wmoMeshCache = new Dictionary<string, MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>?>(StringComparer.OrdinalIgnoreCase);

        // NOTE: WorldScene applies an extra 180° Z correction for its renderer pipeline.
        // For GLB export we omit that correction, since we are not doing the same winding flips.

        foreach (var p in result.MddfPlacements)
        {
            if ((uint)p.NameIndex >= (uint)mdxNames.Count) continue;

            string modelPath = mdxNames[p.NameIndex];
            string key = WorldAssetManager.NormalizeKey(modelPath);

            if (!mdxMeshCache.TryGetValue(key, out var mdxMesh))
            {
                mdxMesh = TryLoadMdxMesh(modelPath, dataSource);
                mdxMeshCache[key] = mdxMesh;
            }

            if (mdxMesh == null) continue;

            float scale = p.Scale > 0 ? p.Scale : 1.0f;

            // Same axis swap/negation as WorldScene for Alpha MDDF.
            float rx = -p.Rotation.Y * MathF.PI / 180f;
            float ry = -p.Rotation.X * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            var transformZup = Matrix4x4.CreateScale(scale)
                * Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);

            scene.AddRigidMesh(mdxMesh, ConvertTransformZupToYup(transformZup));
        }

        foreach (var p in result.ModfPlacements)
        {
            if ((uint)p.NameIndex >= (uint)wmoNames.Count) continue;

            string modelPath = wmoNames[p.NameIndex];
            string key = WorldAssetManager.NormalizeKey(modelPath);

            if (!wmoMeshCache.TryGetValue(key, out var wmoMesh))
            {
                wmoMesh = TryLoadWmoMesh(modelPath, dataSource);
                wmoMeshCache[key] = wmoMesh;
            }

            if (wmoMesh == null) continue;

            float rx = p.Rotation.X * MathF.PI / 180f;
            float ry = p.Rotation.Y * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            var transformZup = Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);

            scene.AddRigidMesh(wmoMesh, ConvertTransformZupToYup(transformZup));
        }
    }

    private static Matrix4x4 ConvertTransformZupToYup(Matrix4x4 zupTransform)
    {
        // If vertices are converted Z-up → Y-up at mesh build time, transforms must be conjugated:
        // T_yup = C * T_zup * C^{-1}.
        Matrix4x4 c = ZupToYupMatrix;
        Matrix4x4 cInv = Matrix4x4.Transpose(c);
        return c * zupTransform * cInv;
    }

    private static MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>? BuildTerrainTileMesh(
        int tileX,
        int tileY,
        IReadOnlyList<TerrainChunkData> chunks,
        string mapName,
        IDataSource dataSource,
        Md5TranslateIndex? md5Index)
    {
        // Material: minimap tile if available.
        var terrainMat = new MaterialBuilder("terrain");
        terrainMat.WithDoubleSide(true);

        byte[]? minimapPng = TryLoadMinimapPngBytes(dataSource, md5Index, mapName, tileX, tileY);
        if (minimapPng != null)
        {
            terrainMat.WithBaseColor(new SharpGLTF.Memory.MemoryImage(minimapPng));
        }
        else
        {
            terrainMat.WithBaseColor(new Vector4(0.65f, 0.65f, 0.65f, 1.0f));
        }

        var mesh = new MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>($"terrain_{tileX}_{tileY}");
        var prim = mesh.UsePrimitive(terrainMat);

        // Build geometry as one mesh (all chunks).
        const int vertsPerChunk = 145;

        // Precompute vertices per chunk.
        var chunkVertices = new VERTEX[chunks.Count][];
        for (int chunkIndex = 0; chunkIndex < chunks.Count; chunkIndex++)
        {
            var chunk = chunks[chunkIndex];
            var vertices = new VERTEX[vertsPerChunk];

            for (int i = 0; i < vertsPerChunk; i++)
            {
                GetVertexPosition(i, out int row, out int col, out bool isInner);

                float cellSize = WoWConstants.ChunkSize / 16f;
                float subCellSize = cellSize / 8f;

                float localX, localY;
                if (!isInner)
                {
                    localX = col * subCellSize;
                    localY = (row / 2) * subCellSize;
                }
                else
                {
                    localX = (col + 0.5f) * subCellSize;
                    localY = (row / 2 + 0.5f) * subCellSize;
                }

                float z = (i < chunk.Heights.Length) ? chunk.Heights[i] : 0f;

                // Renderer coords (Z-up)
                float wx = chunk.WorldPosition.X - localY;
                float wy = chunk.WorldPosition.Y - localX;

                // Convert to glTF Y-up
                var pos = ZupToYup(wx, wy, z);

                var n = (i < chunk.Normals.Length) ? chunk.Normals[i] : Vector3.UnitZ;
                var normal = Vector3.Normalize(ZupToYup(n.X, n.Y, n.Z));

                // UVs in tile space for minimap mapping.
                float uChunk = !isInner ? col / 8f : (col + 0.5f) / 8f;
                float vChunk = !isInner ? (row / 2) / 8f : (row / 2 + 0.5f) / 8f;

                float uTile = (chunk.ChunkX + uChunk) / 16f;
                float vTile = 1f - (chunk.ChunkY + vChunk) / 16f;

                vertices[i] = new VERTEX(
                    new VertexPositionNormal(pos, normal),
                    new VertexTexture1(new Vector2(uTile, vTile)));
            }

            chunkVertices[chunkIndex] = vertices;
        }

        for (int chunkIndex = 0; chunkIndex < chunks.Count; chunkIndex++)
        {
            var chunk = chunks[chunkIndex];
            var indices = BuildChunkIndices(chunk.HoleMask);
            var verts = chunkVertices[chunkIndex];

            for (int t = 0; t + 2 < indices.Length; t += 3)
            {
                prim.AddTriangle(
                    verts[indices[t + 0]],
                    verts[indices[t + 1]],
                    verts[indices[t + 2]]);
            }
        }

        return mesh;
    }

    private static Vector3 ZupToYup(float x, float y, float z) => new(x, z, -y);

    private static byte[]? TryLoadMinimapPngBytes(
        IDataSource dataSource,
        Md5TranslateIndex? md5Index,
        string mapName,
        int tileX,
        int tileY)
    {
        string plainPath = MinimapService.GetMinimapTilePath(mapName, tileX, tileY);

        byte[]? data = null;

        if (md5Index != null)
        {
            var normalized = md5Index.Normalize(plainPath);
            if (md5Index.PlainToHash.TryGetValue(normalized, out string? hashedPath))
            {
                data = dataSource.ReadFile(hashedPath) ?? dataSource.ReadFile(hashedPath.Replace('/', '\\'));
            }
        }

        data ??= dataSource.ReadFile(plainPath) ?? dataSource.ReadFile(plainPath.Replace('/', '\\'));

        if (data == null || data.Length == 0)
            return null;

        // Minimap tiles are typically BLP.
        return BlpToPngBytes(data);
    }

    private static byte[]? BlpToPngBytes(byte[] blpData)
    {
        try
        {
            using var ms = new MemoryStream(blpData);
            using var blp = new BlpFile(ms);
            using var bmp = blp.GetBitmap(0);

            int w = bmp.Width, h = bmp.Height;
            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var lockBits = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);

            byte[] pixels = new byte[w * h * 4];
            System.Runtime.InteropServices.Marshal.Copy(lockBits.Scan0, pixels, 0, pixels.Length);
            bmp.UnlockBits(lockBits);

            // BGRA → RGBA
            for (int i = 0; i < pixels.Length; i += 4)
            {
                (pixels[i], pixels[i + 2]) = (pixels[i + 2], pixels[i]);
            }

            using var image = Image.LoadPixelData<Rgba32>(pixels, w, h);
            using var output = new MemoryStream();
            image.SaveAsPng(output);
            return output.ToArray();
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MapGlbExporter] BLP→PNG decode failed: {ex.Message}");
            return null;
        }
    }

    private static MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>? TryLoadMdxMesh(string modelPath, IDataSource dataSource)
    {
        try
        {
            byte[]? mdxData = dataSource.ReadFile(modelPath);
            if (mdxData == null)
            {
                // MDL/MDX swap
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

            var mesh = new MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>($"mdx_{Path.GetFileNameWithoutExtension(modelPath)}");

            for (int gi = 0; gi < mdx.Geosets.Count; gi++)
            {
                var geoset = mdx.Geosets[gi];
                if (geoset.Vertices.Count == 0 || geoset.Indices.Count == 0)
                    continue;

                var matBuilder = new MaterialBuilder($"mdx_mat_{gi}");
                matBuilder.WithDoubleSide(true);

                byte[]? pngBytes = TryResolveMdxTexturePngBytes(mdx, geoset, dataSource);
                if (pngBytes != null)
                    matBuilder.WithBaseColor(new SharpGLTF.Memory.MemoryImage(pngBytes));
                else
                    matBuilder.WithBaseColor(new Vector4(0.75f, 0.75f, 0.75f, 1.0f));

                var prim = mesh.UsePrimitive(matBuilder);

                bool hasNormals = geoset.Normals.Count == geoset.Vertices.Count;
                bool hasUVs = geoset.TexCoords.Count == geoset.Vertices.Count;

                for (int t = 0; t + 2 < geoset.Indices.Count; t += 3)
                {
                    int i0 = geoset.Indices[t];
                    int i1 = geoset.Indices[t + 1];
                    int i2 = geoset.Indices[t + 2];

                    if ((uint)i0 >= (uint)geoset.Vertices.Count || (uint)i1 >= (uint)geoset.Vertices.Count || (uint)i2 >= (uint)geoset.Vertices.Count)
                        continue;

                    var v0 = MakeMdxVertex(geoset, i0, hasNormals, hasUVs);
                    var v1 = MakeMdxVertex(geoset, i1, hasNormals, hasUVs);
                    var v2 = MakeMdxVertex(geoset, i2, hasNormals, hasUVs);

                    prim.AddTriangle(v0, v1, v2);
                }
            }

            return mesh;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MapGlbExporter] MDX load failed: {modelPath} — {ex.Message}");
            return null;
        }
    }

    private static byte[]? TryResolveMdxTexturePngBytes(MdxFile mdx, MdlGeoset geoset, IDataSource dataSource)
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

        byte[]? blpData = dataSource.ReadFile(tex.Path) ?? dataSource.ReadFile(tex.Path.Replace('/', '\\'));
        if (blpData == null || blpData.Length == 0)
            return null;

        return BlpToPngBytes(blpData);
    }

    private static VERTEX MakeMdxVertex(MdlGeoset geoset, int idx, bool hasNormals, bool hasUVs)
    {
        var pos = geoset.Vertices[idx];
        var position = ZupToYup(pos.X, pos.Y, pos.Z);

        var normal = hasNormals
            ? Vector3.Normalize(ZupToYup(geoset.Normals[idx].X, geoset.Normals[idx].Y, geoset.Normals[idx].Z))
            : Vector3.UnitY;

        var uv = hasUVs
            ? new Vector2(geoset.TexCoords[idx].U, geoset.TexCoords[idx].V)
            : Vector2.Zero;

        return new VERTEX(
            new VertexPositionNormal(position, normal),
            new VertexTexture1(uv));
    }

    private static MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>? TryLoadWmoMesh(string modelPath, IDataSource dataSource)
    {
        try
        {
            byte[]? data = dataSource.ReadFile(modelPath);
            if (data == null || data.Length == 0)
                return null;

            // Detect version similarly to WorldAssetManager.
            int version = DetectWmoVersion(data);

            WmoV14ToV17Converter.WmoV14Data wmo;

            if (version >= 17)
            {
                var dir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? "";
                var baseName = Path.GetFileNameWithoutExtension(modelPath);

                var groupBytesList = new List<byte[]>();
                for (int gi = 0; gi < 512; gi++)
                {
                    var groupName = $"{baseName}_{gi:D3}.wmo";
                    var groupPath = string.IsNullOrEmpty(dir) ? groupName : $"{dir}\\{groupName}";
                    var groupBytes = dataSource.ReadFile(groupPath);
                    if (groupBytes == null || groupBytes.Length == 0) break;
                    groupBytesList.Add(groupBytes);
                }

                var v17Parser = new WmoV17ToV14Converter();
                wmo = v17Parser.ParseV17ToModel(data, groupBytesList);
            }
            else
            {
                // v14/v16: use existing converter via temp file, then parse group files if needed.
                string tmpPath = Path.Combine(Path.GetTempPath(), $"wmo_{Guid.NewGuid():N}.tmp");
                try
                {
                    File.WriteAllBytes(tmpPath, data);
                    var converter = new WmoV14ToV17Converter();
                    wmo = converter.ParseWmoV14(tmpPath);

                    if (wmo.Groups.Count == 0 && wmo.GroupCount > 0)
                    {
                        var wmoDir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? "";
                        var wmoBase = Path.GetFileNameWithoutExtension(modelPath);
                        for (int gi = 0; gi < wmo.GroupCount; gi++)
                        {
                            var groupName = $"{wmoBase}_{gi:D3}.wmo";
                            var groupPath = string.IsNullOrEmpty(wmoDir) ? groupName : $"{wmoDir}\\{groupName}";
                            var groupBytes = dataSource.ReadFile(groupPath);
                            if (groupBytes != null && groupBytes.Length > 0)
                                converter.ParseGroupFile(groupBytes, wmo, gi);
                        }
                        for (int gi = 0; gi < wmo.Groups.Count; gi++)
                        {
                            if (wmo.Groups[gi].Name == null)
                                wmo.Groups[gi].Name = $"group_{gi}";
                        }
                    }
                }
                finally
                {
                    try { File.Delete(tmpPath); } catch { }
                }
            }

            return BuildWmoMesh(wmo, modelPath, dataSource);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[MapGlbExporter] WMO load failed: {modelPath} — {ex.Message}");
            return null;
        }
    }

    private static MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty> BuildWmoMesh(
        WmoV14ToV17Converter.WmoV14Data wmo,
        string modelPath,
        IDataSource dataSource)
    {
        string baseName = Path.GetFileNameWithoutExtension(modelPath);
        var mesh = new MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>($"wmo_{baseName}");

        var defaultMat = new MaterialBuilder("wmo_default");
        defaultMat.WithBaseColor(new Vector4(0.6f, 0.6f, 0.6f, 1.0f));
        defaultMat.WithDoubleSide(true);

        var materials = new Dictionary<byte, MaterialBuilder>();
        for (int m = 0; m < wmo.Materials.Count; m++)
        {
            var wmoMat = wmo.Materials[m];
            var matBuilder = new MaterialBuilder($"wmo_mat_{m}");
            matBuilder.WithDoubleSide(true);

            // Best-effort texture resolve via data source.
            if (!string.IsNullOrEmpty(wmoMat.Texture1Name))
            {
                byte[]? png = TryResolveWmoTexturePngBytes(wmoMat.Texture1Name, dataSource);
                if (png != null)
                    matBuilder.WithBaseColor(new SharpGLTF.Memory.MemoryImage(png));
                else
                    matBuilder.WithBaseColor(DistinctColor(m));
            }
            else
            {
                matBuilder.WithBaseColor(DistinctColor(m));
            }

            materials[(byte)m] = matBuilder;
        }

        // material → primitive cache so we don't call UsePrimitive repeatedly.
        var primCache = new Dictionary<MaterialBuilder, MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>.PrimitiveBuilder>();

        for (int gi = 0; gi < wmo.Groups.Count; gi++)
        {
            var group = wmo.Groups[gi];
            if (group.Vertices.Count == 0 || group.Indices.Count == 0)
                continue;

            var normals = GenerateNormals(group);
            bool hasUVs = group.UVs.Count == group.Vertices.Count;

            int faceCount = group.Indices.Count / 3;
            for (int f = 0; f < faceCount; f++)
            {
                int idx = f * 3;
                int i0 = group.Indices[idx];
                int i1 = group.Indices[idx + 1];
                int i2 = group.Indices[idx + 2];

                if ((uint)i0 >= (uint)group.Vertices.Count || (uint)i1 >= (uint)group.Vertices.Count || (uint)i2 >= (uint)group.Vertices.Count)
                    continue;

                byte matId = f < group.FaceMaterials.Count ? group.FaceMaterials[f] : (byte)0;
                if (matId == 0xFF)
                    continue;

                var mat = materials.GetValueOrDefault(matId, defaultMat);

                if (!primCache.TryGetValue(mat, out var prim))
                {
                    prim = mesh.UsePrimitive(mat);
                    primCache[mat] = prim;
                }

                var v0 = MakeWmoVertex(group, i0, normals, hasUVs);
                var v1 = MakeWmoVertex(group, i1, normals, hasUVs);
                var v2 = MakeWmoVertex(group, i2, normals, hasUVs);

                prim.AddTriangle(v0, v1, v2);
            }
        }

        return mesh;
    }

    private static Vector4 DistinctColor(int i)
    {
        float r = ((i * 67 + 13) % 255) / 255f;
        float g = ((i * 131 + 7) % 255) / 255f;
        float b = ((i * 43 + 29) % 255) / 255f;
        return new Vector4(r, g, b, 1.0f);
    }

    private static byte[]? TryResolveWmoTexturePngBytes(string textureName, IDataSource dataSource)
    {
        byte[]? blp = dataSource.ReadFile(textureName) ?? dataSource.ReadFile(textureName.Replace('/', '\\'));
        if (blp == null || blp.Length == 0)
            return null;
        return BlpToPngBytes(blp);
    }

    private static VERTEX MakeWmoVertex(
        WmoV14ToV17Converter.WmoGroupData group,
        int idx,
        List<Vector3> normals,
        bool hasUVs)
    {
        var pos = group.Vertices[idx];
        var position = ZupToYup(pos.X, pos.Y, pos.Z);

        var normal = idx < normals.Count
            ? Vector3.Normalize(ZupToYup(normals[idx].X, normals[idx].Y, normals[idx].Z))
            : Vector3.UnitY;

        var uv = hasUVs ? group.UVs[idx] : Vector2.Zero;

        return new VERTEX(
            new VertexPositionNormal(position, normal),
            new VertexTexture1(uv));
    }

    private static List<Vector3> GenerateNormals(WmoV14ToV17Converter.WmoGroupData group)
    {
        var normals = new Vector3[group.Vertices.Count];
        for (int i = 0; i + 2 < group.Indices.Count; i += 3)
        {
            int i0 = group.Indices[i], i1 = group.Indices[i + 1], i2 = group.Indices[i + 2];
            if ((uint)i0 >= (uint)group.Vertices.Count || (uint)i1 >= (uint)group.Vertices.Count || (uint)i2 >= (uint)group.Vertices.Count)
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

    private static int DetectWmoVersion(byte[] data)
    {
        if (data.Length < 12) return 0;
        string magic = System.Text.Encoding.ASCII.GetString(data, 0, 4);
        string reversed = new string(magic.Reverse().ToArray());

        if (magic == "MOMO" || reversed == "MOMO") return 14;

        if (magic == "MVER" || reversed == "MVER")
        {
            uint size = BitConverter.ToUInt32(data, 4);
            if (size >= 4 && data.Length >= 12)
                return (int)BitConverter.ToUInt32(data, 8);
        }
        return 0;
    }

    private static void GetVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        // Matches TerrainTileMeshBuilder.GetVertexPosition.
        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;

        for (int r = 0; r < 17; r++)
        {
            int rowSize = (r % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = r;
                col = remaining;
                isInner = (r % 2 != 0);
                return;
            }
            remaining -= rowSize;
        }
    }

    private static int OuterIndex(int outerRow, int outerCol) => outerRow * 17 + outerCol;
    private static int InnerIndex(int innerRow, int innerCol) => innerRow * 17 + 9 + innerCol;

    private static ushort[] BuildChunkIndices(int holeMask)
    {
        // Matches TerrainTileMeshBuilder.BuildIndices.
        var indices = new List<ushort>(256 * 3);

        for (int cellY = 0; cellY < 8; cellY++)
        {
            for (int cellX = 0; cellX < 8; cellX++)
            {
                if (holeMask != 0)
                {
                    int holeX = cellX / 2;
                    int holeY = cellY / 2;
                    int holeBit = 1 << (holeY * 4 + holeX);
                    if ((holeMask & holeBit) != 0)
                        continue;
                }

                ushort tl = (ushort)OuterIndex(cellY, cellX);
                ushort tr = (ushort)OuterIndex(cellY, cellX + 1);
                ushort bl = (ushort)OuterIndex(cellY + 1, cellX);
                ushort br = (ushort)OuterIndex(cellY + 1, cellX + 1);
                ushort center = (ushort)InnerIndex(cellY, cellX);

                indices.Add(center);
                indices.Add(tr);
                indices.Add(tl);

                indices.Add(center);
                indices.Add(br);
                indices.Add(tr);

                indices.Add(center);
                indices.Add(bl);
                indices.Add(br);

                indices.Add(center);
                indices.Add(tl);
                indices.Add(bl);
            }
        }

        return indices.ToArray();
    }
}
