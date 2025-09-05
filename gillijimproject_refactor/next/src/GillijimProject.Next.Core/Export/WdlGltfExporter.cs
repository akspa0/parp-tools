using System;
using System.IO;
using System.Numerics;
using GillijimProject.Next.Core.Domain;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Scenes;

namespace GillijimProject.Next.Core.Export;

public static class WdlGltfExporter
{
    private const double CellWorldSize = 533.3333333333 / 16.0; // yards per height cell in ADT terms
    public sealed record ExportOptions(
        double Scale = 1.0,
        bool SkipHoles = true,
        bool NormalizeWorld = true,
        double HeightScale = 1.0
    );

    public sealed record ExportStats(
        int TilesExported,
        int VerticesApprox,
        int FacesWritten
    );

    public static ExportStats ExportPerTile(Wdl wdl, string tilesDir, ExportOptions? options = null)
    {
        options ??= new ExportOptions();
        Directory.CreateDirectory(tilesDir);

        int tiles = 0, faces = 0, vApprox = 0;
        double xyScale = (options.NormalizeWorld ? CellWorldSize : 1.0) * options.Scale;
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                var t = wdl.Tiles[y, x];
                if (t is null) continue;

                var scene = new SceneBuilder();
                var mesh = BuildTileMesh(t, xyScale, options.HeightScale, options.SkipHoles, out int vCount, out int fCount);
                scene.AddRigidMesh(mesh, Matrix4x4.Identity);

                var model = scene.ToGltf2();
                var outPath = Path.Combine(tilesDir, $"tile_{x}_{y}.glb");
                model.SaveGLB(outPath);

                tiles++;
                faces += fCount;
                vApprox += vCount;
            }
        }
        return new ExportStats(tiles, vApprox, faces);
    }

    public static ExportStats ExportMerged(Wdl wdl, string mergedGlbPath, ExportOptions? options = null)
    {
        options ??= new ExportOptions();
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(mergedGlbPath)) ?? ".");

        int tiles = 0, faces = 0, vApprox = 0;
        var scene = new SceneBuilder();
        double xyScale = (options.NormalizeWorld ? CellWorldSize : 1.0) * options.Scale;

        for (int ty = 0; ty < 64; ty++)
        {
            for (int tx = 0; tx < 64; tx++)
            {
                var t = wdl.Tiles[ty, tx];
                if (t is null) continue;

                var mesh = BuildTileMesh(t, xyScale, options.HeightScale, options.SkipHoles, out int vCount, out int fCount);
                float baseX = (float)(tx * 16 * xyScale);
                float baseZ = (float)(ty * 16 * xyScale);
                var xform = Matrix4x4.CreateTranslation(baseX, 0f, baseZ);
                scene.AddRigidMesh(mesh, xform);

                tiles++;
                faces += fCount;
                vApprox += vCount;
            }
        }

        var model = scene.ToGltf2();
        model.SaveGLB(mergedGlbPath);
        return new ExportStats(tiles, vApprox, faces);
    }

    private static MeshBuilder<VertexPositionNormal, VertexEmpty, VertexEmpty> BuildTileMesh(WdlTile tile, double xyScale, double heightScale, bool skipHoles, out int verticesApprox, out int faces)
    {
        var mat = new MaterialBuilder("mat").WithMetallicRoughnessShader();
        var mesh = new MeshBuilder<VertexPositionNormal, VertexEmpty, VertexEmpty>("tile");
        var prim = mesh.UsePrimitive(mat);
        faces = 0;

        // Precompute normals per 17x17 vertex
        var normals = new Vector3[17, 17];
        for (int j = 0; j <= 16; j++)
        {
            for (int i = 0; i <= 16; i++)
            {
                float hL = tile.Height17[j, Math.Max(0, i - 1)];
                float hR = tile.Height17[j, Math.Min(16, i + 1)];
                float hU = tile.Height17[Math.Max(0, j - 1), i];
                float hD = tile.Height17[Math.Min(16, j + 1), i];
                var dx = new Vector3((float)(2 * xyScale), (hR - hL) * (float)heightScale, 0);
                var dz = new Vector3(0, (hD - hU) * (float)heightScale, (float)(2 * xyScale));
                var n = Vector3.Cross(dz, dx);
                if (n.LengthSquared() < 1e-6f) n = new Vector3(0, 1, 0);
                else n = Vector3.Normalize(n);
                normals[j, i] = n;
            }
        }

        static VertexPositionNormal V(int i, int j, double xy, double hScale, WdlTile t, Vector3[,] ns)
        {
            var pos = new Vector3((float)(i * xy), t.Height17[j, i] * (float)hScale, (float)(j * xy));
            var nrm = ns[j, i];
            return new VertexPositionNormal(pos, nrm);
        }

        for (int j = 0; j < 16; j++)
        {
            for (int i = 0; i < 16; i++)
            {
                if (skipHoles && tIsHole(tile, j, i)) continue;
                var v00 = V(i, j, xyScale, heightScale, tile, normals);
                var v10 = V(i + 1, j, xyScale, heightScale, tile, normals);
                var v01 = V(i, j + 1, xyScale, heightScale, tile, normals);
                var v11 = V(i + 1, j + 1, xyScale, heightScale, tile, normals);
                prim.AddTriangle(v00, v10, v11);
                prim.AddTriangle(v00, v11, v01);
                faces += 2;
            }
        }

        verticesApprox = 17 * 17; // approximate, independent of holes
        return mesh;

        static bool tIsHole(WdlTile t, int y, int x) => t.IsHole(y, x);
    }
}
